"""Deterministic bounded-Gini CSR row-length generation."""

from __future__ import annotations

import numpy as np


MIN_DEGREE_RATIO = 0.2
MAX_DEGREE_RATIO = 30.0
GINI_TOLERANCE = 0.002


def gini_coefficient(values) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("row lengths must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("row lengths must be finite and non-negative")
    total = float(values.sum())
    if total == 0.0:
        return 0.0
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    return float(
        2.0 * np.dot(ranks, values) / (values.size * total)
        - (values.size + 1.0) / values.size
    )


def make_gini_row_lengths(
    n_rows: int,
    mean_degree: int,
    target_gini: float,
    *,
    seed: int,
    tolerance: float = GINI_TOLERANCE,
) -> tuple[np.ndarray, float]:
    n_rows = int(n_rows)
    mean_degree = int(mean_degree)
    target_gini = float(target_gini)
    tolerance = float(tolerance)
    if n_rows <= 1 or mean_degree < 1 or mean_degree > n_rows:
        raise ValueError("invalid row count or mean row length")
    if not np.isfinite(target_gini) or not 0.0 < target_gini < 1.0:
        raise ValueError("target Gini must be finite and in (0, 1)")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Gini tolerance must be finite and non-negative")

    low = max(1, int(np.floor(MIN_DEGREE_RATIO * mean_degree)))
    high = min(n_rows, int(np.ceil(MAX_DEGREE_RATIO * mean_degree)))
    if low >= mean_degree or high <= mean_degree:
        raise ValueError(
            f"target Gini {target_gini:g} is not attainable with exact nnz "
            f"and row lengths in [{low}, {high}]"
        )

    ranks = (np.arange(n_rows, dtype=np.float64) + 0.5) / n_rows

    def curve(beta: float) -> np.ndarray:
        raw = ranks ** -beta
        scale_low = 0.0
        scale_high = 1.0
        while np.clip(scale_high * raw, low, high).mean() < mean_degree:
            scale_high *= 2.0
        for _ in range(52):
            scale_mid = (scale_low + scale_high) * 0.5
            if np.clip(scale_mid * raw, low, high).mean() < mean_degree:
                scale_low = scale_mid
            else:
                scale_high = scale_mid
        return np.clip(
            (scale_low + scale_high) * 0.5 * raw,
            low,
            high,
        )

    beta_low = 0.0
    beta_high = 1.0
    high_curve = curve(beta_high)
    while gini_coefficient(high_curve) < target_gini:
        beta_high *= 2.0
        if beta_high > 32.0:
            raise ValueError(
                f"target Gini {target_gini:g} is not attainable by the "
                "bounded row-length distribution"
            )
        high_curve = curve(beta_high)

    continuous = high_curve
    for _ in range(48):
        beta_mid = (beta_low + beta_high) * 0.5
        continuous = curve(beta_mid)
        if gini_coefficient(continuous) < target_gini:
            beta_low = beta_mid
        else:
            beta_high = beta_mid
    continuous = curve((beta_low + beta_high) * 0.5)

    row_lengths = np.floor(continuous).astype(np.int64)
    target_nnz = n_rows * mean_degree
    remainder = target_nnz - int(row_lengths.sum())
    if remainder < 0:
        raise RuntimeError("integer row lengths exceeded the exact nnz target")
    if remainder:
        fractions = continuous - row_lengths
        eligible = np.flatnonzero(row_lengths < high)
        if remainder > eligible.size:
            raise ValueError("exact nnz is not attainable within row-length bounds")
        if remainder == eligible.size:
            chosen = eligible
        else:
            chosen = eligible[
                np.argpartition(fractions[eligible], -remainder)[-remainder:]
            ]
        row_lengths[chosen] += 1

    actual_gini = gini_coefficient(row_lengths)
    if (
        int(row_lengths.min()) < 1
        or int(row_lengths.sum()) != target_nnz
        or abs(actual_gini - target_gini) > tolerance
    ):
        raise ValueError(
            f"target Gini {target_gini:g} is not attainable after integerization; "
            f"actual Gini is {actual_gini:.9f}"
        )
    np.random.default_rng(seed).shuffle(row_lengths)
    return np.ascontiguousarray(row_lengths, dtype=np.int32), actual_gini
