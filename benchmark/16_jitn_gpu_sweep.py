#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainevent import jitn, jitn_p


@dataclass(frozen=True)
class Case:
    shape: Tuple[int, int]
    prob: float
    corder: bool
    transpose: bool
    backend: str


def _parse_shapes(raw: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for chunk in raw.split(","):
        part = chunk.strip().lower()
        if not part:
            continue
        m_s, n_s = part.split("x", maxsplit=1)
        out.append((int(m_s), int(n_s)))
    return out


def _parse_probs(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _pick_repeats(shape: Tuple[int, int], repeats_override: int) -> int:
    if repeats_override > 0:
        return repeats_override
    elems = shape[0] * shape[1]
    if elems >= 8_000_000:
        return 8
    if elems >= 2_000_000:
        return 12
    return 20


def _bench_case(case: Case, warmup: int, repeats: int) -> Tuple[float, float, float]:
    m, n = case.shape

    @jax.jit
    def _run():
        return jitn(
            1.5,
            0.15,
            case.prob,
            123,
            shape=(m, n),
            transpose=case.transpose,
            corder=case.corder,
            backend=case.backend,
        )

    # compile once
    _run().block_until_ready()
    for _ in range(warmup):
        _run().block_until_ready()

    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _run().block_until_ready()
        times_ms.append((time.perf_counter() - t0) * 1e3)

    med = statistics.median(times_ms)
    p10 = statistics.quantiles(times_ms, n=10, method="inclusive")[0]
    p90 = statistics.quantiles(times_ms, n=10, method="inclusive")[8]
    return med, p10, p90


def _available_gpu_backends() -> Sequence[str]:
    platform = jax.default_backend()
    available = set(jitn_p.available_backends(platform))
    return [b for b in ("warp", "pallas") if b in available]


def main():
    parser = argparse.ArgumentParser(description="Reproducible GPU throughput sweep for jitn kernels.")
    parser.add_argument(
        "--shapes",
        type=str,
        default="512x512,1024x1024,2048x2048,4096x1024,1024x4096",
        help="Comma-separated list like 512x512,2048x1024",
    )
    parser.add_argument(
        "--probs",
        type=str,
        default="0.02,0.1,0.25",
        help="Comma-separated connection probabilities.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations after compilation.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="Benchmark repeats per case. 0 means shape-adaptive repeats.",
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Include transpose=True runs (default: transpose=False only).",
    )
    args = parser.parse_args()

    shapes = _parse_shapes(args.shapes)
    probs = _parse_probs(args.probs)
    backends = _available_gpu_backends()
    if not backends:
        raise RuntimeError("No GPU jitn backend available from {'warp', 'pallas'} on this platform.")

    transpose_options = [False, True] if args.transpose else [False]

    print(f"# platform={jax.default_backend()} backends={backends}")
    print("shape,prob,corder,transpose,backend,repeats,median_ms,p10_ms,p90_ms,est_dense_gelem_s,est_nnz_gelem_s")

    results = {}
    for shape in shapes:
        for prob in probs:
            for corder in (True, False):
                for transpose in transpose_options:
                    for backend in backends:
                        case = Case(
                            shape=shape,
                            prob=prob,
                            corder=corder,
                            transpose=transpose,
                            backend=backend,
                        )
                        repeats = _pick_repeats(shape, args.repeats)
                        med, p10, p90 = _bench_case(case, warmup=args.warmup, repeats=repeats)
                        dense_gelem_s = (shape[0] * shape[1]) / (med * 1e-3) / 1e9
                        nnz_gelem_s = (shape[0] * shape[1] * prob) / (med * 1e-3) / 1e9
                        print(
                            f"{shape[0]}x{shape[1]},{prob},{corder},{transpose},{backend},{repeats},"
                            f"{med:.6f},{p10:.6f},{p90:.6f},{dense_gelem_s:.6f},{nnz_gelem_s:.6f}"
                        )
                        results[(shape, prob, corder, transpose, backend)] = med

    if "warp" in backends and "pallas" in backends:
        print("\n# summary: pallas_vs_warp_speedup ( >1 means pallas faster )")
        print("shape,prob,corder,transpose,pallas_over_warp")
        for shape in shapes:
            for prob in probs:
                for corder in (True, False):
                    for transpose in transpose_options:
                        p = results[(shape, prob, corder, transpose, "pallas")]
                        w = results[(shape, prob, corder, transpose, "warp")]
                        print(f"{shape[0]}x{shape[1]},{prob},{corder},{transpose},{w / p:.4f}")


if __name__ == "__main__":
    main()
